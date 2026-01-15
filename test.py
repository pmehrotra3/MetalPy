from MetalPy import MetalPy, MetalArray
import numpy as np
import math
from math import erf, sqrt
import array
import os
import sys
from datetime import datetime

gpu = MetalPy()

# ==========================================================
# Helper creators
# ==========================================================

def rand_vec(n, scale=100.0):
    return (scale * np.random.rand(n)).astype(np.float32)

def rand_mat(rows, cols, scale=10.0):
    return (scale * np.random.rand(rows * cols)).astype(np.float32)


# ==========================================================
# CATEGORY 1: ELEMENTWISE MATH OPERATIONS
# ==========================================================

def test1():  # add
    
    a = rand_vec(100)
    b = rand_vec(100)
    
    v1 = gpu.add(gpu.array(a.tolist()), gpu.array(b.tolist())).to_array()
    v2 = a + b
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "add() failed"


def test2():  # sub
    
    a = rand_vec(100)
    b = rand_vec(100)
    
    v1 = gpu.sub(gpu.array(a.tolist()), gpu.array(b.tolist())).to_array()
    v2 = a - b
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "sub() failed"


def test3():  # multiply
    
    a = rand_vec(100)
    b = rand_vec(100)
    
    v1 = gpu.multiply(gpu.array(a.tolist()), gpu.array(b.tolist())).to_array()
    v2 = a * b
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "multiply() failed"


def test4():  # divide
    
    a = rand_vec(100)
    b = rand_vec(100) + 1e-3  # avoid zero
    
    v1 = gpu.divide(gpu.array(a.tolist()), gpu.array(b.tolist())).to_array()
    v2 = a / b
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "divide() failed"


def test5():  # negate
    
    a = rand_vec(100) - 50.0
    
    v1 = gpu.negate(gpu.array(a.tolist())).to_array()
    v2 = -a
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "negate() failed"


def test6():  # abs
    
    a = rand_vec(100) - 50.0
    
    v1 = gpu.abs(gpu.array(a.tolist())).to_array()
    v2 = np.abs(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "abs() failed"


def test7():  # pow
    
    a = np.abs(rand_vec(100)) + 1e-3
    scalar = float(np.random.uniform(0.3, 3.0))  # random scalar
    
    v1 = gpu.pow(gpu.array(a.tolist()), scalar).to_array()
    v2 = a ** scalar
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "pow() failed"


def test8():  # square
    
    a = rand_vec(100)
    
    v1 = gpu.square(gpu.array(a.tolist())).to_array()
    v2 = a ** 2
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "square() failed"


def test9():  # sqrt
    
    a = np.abs(rand_vec(100)) + 1e-3
    
    v1 = gpu.sqrt(gpu.array(a.tolist())).to_array()
    v2 = np.sqrt(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "sqrt() failed"


def test10():  # exp
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)  # [-2,2]
    
    v1 = gpu.exp(gpu.array(a.tolist())).to_array()
    v2 = np.exp(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "exp() failed"


def test11():  # log
    
    a = np.abs(rand_vec(100)) + 1e-3
    
    v1 = gpu.log(gpu.array(a.tolist())).to_array()
    v2 = np.log(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "log() failed"


def test12():  # sin
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.sin(gpu.array(a.tolist())).to_array()
    v2 = np.sin(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "sin() failed"


def test13():  # cos
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.cos(gpu.array(a.tolist())).to_array()
    v2 = np.cos(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "cos() failed"


def test14():  # tan
    
    a = (np.random.rand(100) * 1.0 - 0.5).astype(np.float32)  # 
    
    v1 = gpu.tan(gpu.array(a.tolist())).to_array()
    v2 = np.tan(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "tan() failed"


def test15():  # asin
    
    a = (np.random.rand(100) * 2.0 - 1.0).astype(np.float32)  # [-1,1]
    
    v1 = gpu.asin(gpu.array(a.tolist())).to_array()
    v2 = np.arcsin(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "asin() failed"


def test16():  # atan2
    
    y = rand_vec(100) - 50.0
    x = rand_vec(100) - 50.0
    
    v1 = gpu.atan2(gpu.array(y.tolist()), gpu.array(x.tolist())).to_array()
    v2 = np.arctan2(y, x)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "atan2() failed"


def test17():  # floor
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.floor(gpu.array(a.tolist())).to_array()
    v2 = np.floor(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "floor() failed"


def test18():  # ceil
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.ceil(gpu.array(a.tolist())).to_array()
    v2 = np.ceil(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "ceil() failed"


def test19():  # sign
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.sign(gpu.array(a.tolist())).to_array()
    v2 = np.sign(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "sign() failed"


def test20():  # clip
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    low, high = -0.5, 1.0
    
    v1 = gpu.clip(gpu.array(a.tolist()), low, high).to_array()
    v2 = np.clip(a, low, high)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "clip() failed"


def test21():  # round
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.round(gpu.array(a.tolist())).to_array()
    v2 = np.round(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "round() failed"


def test22():  # broadcast_add
    
    a = rand_vec(100)
    scalar = np.array([5.0], dtype=np.float32)
    
    v1 = gpu.broadcast_add(gpu.array(a.tolist()), gpu.array(scalar.tolist())).to_array()
    v2 = a + scalar[0]
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "broadcast_add() failed"


def test23():  # broadcast_multiply
    
    a = rand_vec(100)
    scalar = np.array([3.0], dtype=np.float32)
    
    v1 = gpu.broadcast_multiply(gpu.array(a.tolist()), gpu.array(scalar.tolist())).to_array()
    v2 = a * scalar[0]
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "broadcast_multiply() failed"


# ==========================================================
# CATEGORY 2: ACTIVATIONS
# ==========================================================

def test24():  # relu
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.relu(gpu.array(a.tolist())).to_array()
    v2 = np.maximum(a, 0.0)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "relu() failed"


def test25():  # leaky_relu
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    alpha = 0.1
    
    v1 = gpu.leaky_relu(gpu.array(a.tolist()), alpha=alpha).to_array()
    v2 = np.where(a > 0.0, a, alpha * a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "leaky_relu() failed"


def test26():  # sigmoid
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.sigmoid(gpu.array(a.tolist())).to_array()
    v2 = 1.0 / (1.0 + np.exp(-a))
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "sigmoid() failed"


def test27():  # tanh
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.tanh(gpu.array(a.tolist())).to_array()
    v2 = np.tanh(a)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "tanh() failed"


def test28():  # softplus
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    v1 = gpu.softplus(gpu.array(a.tolist())).to_array()
    v2 = np.log1p(np.exp(a))
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "softplus() failed"


def test29():  # swish
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.swish(ga).to_array()
    sig = 1.0 / (1.0 + np.exp(-a))
    v2 = a * sig
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "swish() failed"

def test30():  # gelu
    
    a = (np.random.rand(100) * 4.0 - 2.0).astype(np.float32)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.gelu(ga).to_array()          # Metal result (float32)

    # Compute reference in (effectively) float64, then cast to float32
    erf_vec = np.vectorize(erf)
    v2 = 0.5 * a * (1.0 + erf_vec(a / math.sqrt(2.0)))
    v2 = v2.astype(np.float32)

    assert np.allclose(v1, v2, rtol=3e-4, atol=3e-4), "gelu() failed"


# ==========================================================
# CATEGORY 3: REDUCTIONS
# ==========================================================

def test31():  # sum
    
    a = rand_vec(257)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.sum(ga)
    v2 = float(np.sum(a))
    
    assert math.isclose(v1, v2, rel_tol=1e-5, abs_tol=1e-5), "sum() failed"


def test32():  # product
    
    a = np.random.randint(1, 101, 10).astype(np.float32)  # 100 elements of 1, 2, or 3
    
    ga = gpu.array(a.tolist())
    v1 = gpu.product(ga)
    v2 = float(np.prod(a, dtype=np.float32))
    
    # Product is numerically unstable in FP32
    assert math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.0), "product() failed"


def test33():  # max
    
    a = rand_vec(200)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.max(ga)
    v2 = float(np.max(a))
    
    if not math.isclose(v1, v2, rel_tol=1e-8, abs_tol=1e-8):
        print(f"{name} debug:")
        print(f"  v1 (GPU) = {v1}")
        print(f"  v2 (CPU/NumPy) = {v2}")
        print(f"  abs diff  = {abs(v1 - v2)}")
        print(f"  rel diff  = {abs(v1 - v2) / (abs(v2) if v2 != 0 else 1)}")
        raise AssertionError(f"{name} failed: values not close")
    
    assert math.isclose(v1, v2, rel_tol=0, abs_tol=0), "max() failed"


def test34():  # min
    
    a = rand_vec(200)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.min(ga)
    v2 = float(np.min(a))
    
    assert math.isclose(v1, v2, rel_tol=0, abs_tol=0), "min() failed"


def test35():  # argmax
    
    a = rand_vec(311)
    
    ga = gpu.array(a.tolist())
    idx, val = gpu.argmax(ga)
    idx_np = int(np.argmax(a))
    val_np = float(a[idx_np])
    
    assert idx == idx_np, "argmax() index mismatch"
    assert math.isclose(val, val_np, rel_tol=0, abs_tol=0), "argmax() value mismatch"


def test36():  # argmin
    
    a = rand_vec(311)
    
    ga = gpu.array(a.tolist())
    idx, val = gpu.argmin(ga)
    idx_np = int(np.argmin(a))
    val_np = float(a[idx_np])
    
    assert idx == idx_np, "argmin() index mismatch"
    assert math.isclose(val, val_np, rel_tol=0, abs_tol=0), "argmin() value mismatch"


def test37():  # mean
    
    a = rand_vec(257)
    
    ga = gpu.array(a.tolist())
    v1 = gpu.mean(ga)
    v2 = float(np.mean(a))
    
    assert math.isclose(v1, v2, rel_tol=1e-5, abs_tol=1e-5), "mean() failed"


# ==========================================================
# CATEGORY 4: LINEAR ALGEBRA (Matrix Operations)
# ==========================================================

def test38():  # hadamard
    
    rows, cols = 7, 9
    A = rand_mat(rows, cols)
    B = rand_mat(rows, cols)
    
    gA = gpu.array(A.tolist())
    gB = gpu.array(B.tolist())
    v1 = gpu.hadamard(gA, gB, rows, cols).to_array()
    v2 = (A.reshape(rows, cols) * B.reshape(rows, cols)).ravel()
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "hadamard() failed"


def test39():  # row_sum
    
    rows, cols = 7, 9
    
    A = rand_mat(rows, cols)
    gA = gpu.array(A.tolist())
    v1 = gpu.row_sum(gA, rows, cols).to_array()
    v2 = A.reshape(rows, cols).sum(axis=1)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "row_sum() failed"


def test40():  # col_sum
    
    rows, cols = 7, 9
    
    A = rand_mat(rows, cols)
    gA = gpu.array(A.tolist())
    v1 = gpu.col_sum(gA, rows, cols).to_array()
    v2 = A.reshape(rows, cols).sum(axis=0)
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "col_sum() failed"


def test41():  # row_scale
    
    rows, cols = 7, 9
    
    A = rand_mat(rows, cols)
    s = rand_vec(rows, scale=3.0)
    gA = gpu.array(A.tolist())
    gS = gpu.array(s.tolist())
    v1 = gpu.row_scale(gA, gS, rows, cols).to_array()
    v2 = (s[:, None] * A.reshape(rows, cols)).ravel()
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "row_scale() failed"


def test42():  # col_scale
    
    rows, cols = 7, 9
    A = rand_mat(rows, cols)
    s = rand_vec(cols, scale=3.0)
    
    gA = gpu.array(A.tolist())
    gS = gpu.array(s.tolist())
    v1 = gpu.col_scale(gA, gS, rows, cols).to_array()
    v2 = (A.reshape(rows, cols) * s[None, :]).ravel()
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "col_scale() failed"


def test43():  # transpose
    
    rows, cols = 7, 9
    
    A = rand_mat(rows, cols)
    gA = gpu.array(A.tolist())
    v1 = gpu.transpose(gA, rows, cols).to_array()
    v2 = A.reshape(rows, cols).T.ravel()
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "transpose() failed"


def test44():  # matmul
    
    M, K, N = 5, 4, 6
    A = rand_mat(M, K)
    B = rand_mat(K, N)
    
    gA = gpu.array(A.tolist())
    gB = gpu.array(B.tolist())
    v1 = gpu.matmul(gA, gB, M, K, N).to_array()
    v2 = (A.reshape(M, K) @ B.reshape(K, N)).ravel()
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "matmul() failed"


# ==========================================================
# CATEGORY 5: MISCELLANEOUS
# ==========================================================

def test45():  # slice
    
    a = rand_vec(50)
    
    gA = gpu.array(a.tolist())
    start, end = 5, 25
    v1 = gpu.slice(gA, start, end).to_array()
    v2 = a[start:end]
    
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "slice() failed"


def test46():  # gather
    # IMPORTANT: MetalArray must store typecode, and support array('I') for indices
    source = rand_vec(40)
    indices = np.array([0, 3, 7, 10, 10, 5, 39], dtype=np.uint32)

    g_source = gpu.array(source.tolist())
    idx_arr = array.array('I', indices.tolist())
    g_index = MetalArray(gpu.device, idx_arr)

    v1 = gpu.gather(g_source, g_index).to_array()
    v2 = source[indices]
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-5), "gather() failed"


# ==========================================================
# Main Test Runner
# ==========================================================

if __name__ == "__main__":
    
    os.makedirs("Output", exist_ok=True)
    
    with open(f"Output/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        sys.stdout = f
        
        for name, fn in list(globals().items()):
            if name.startswith("test") and callable(fn):
                print(f"Running {name}...")
                fn()
        print("All tests passed.")
        
        sys.stdout = sys.__stdout__
    
    print("Tests completed! Check Output/ folder.")

    
