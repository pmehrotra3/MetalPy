from    MetalPy import MetalPy
import math

def test_elementwise_math():
    print("\n=== Testing Category 1: Elementwise Math ===")
    gpu = MetalPy()
    
    # Basic operations
    a = gpu.array([1.0, 2.0, 3.0, 4.0])
    b = gpu.array([5.0, 6.0, 7.0, 8.0])
    
    print("a:", a.to_list())
    print("b:", b.to_list())
    
    # Test add
    result = gpu.add(a, b)
    print("add(a, b):", result.to_list())
    assert result.to_list() == [6.0, 8.0, 10.0, 12.0], "Add failed"
    
    # Test sub
    result = gpu.sub(a, b)
    print("sub(a, b):", result.to_list())
    assert result.to_list() == [-4.0, -4.0, -4.0, -4.0], "Sub failed"
    
    # Test multiply
    result = gpu.multiply(a, b)
    print("multiply(a, b):", result.to_list())
    assert result.to_list() == [5.0, 12.0, 21.0, 32.0], "Multiply failed"
    
    # Test divide
    result = gpu.divide(a, b)
    print("divide(a, b):", result.to_list())
    expected = [1.0/5.0, 2.0/6.0, 3.0/7.0, 4.0/8.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Divide failed"
    
    # Test negate
    result = gpu.negate(a)
    print("negate(a):", result.to_list())
    assert result.to_list() == [-1.0, -2.0, -3.0, -4.0], "Negate failed"
    
    # Test abs
    c = gpu.array([-1.0, -2.0, 3.0, -4.0])
    result = gpu.abs(c)
    print("abs([-1, -2, 3, -4]):", result.to_list())
    assert result.to_list() == [1.0, 2.0, 3.0, 4.0], "Abs failed"
    
    # Test pow
    result = gpu.pow(a, 2.0)
    print("pow(a, 2):", result.to_list())
    assert result.to_list() == [1.0, 4.0, 9.0, 16.0], "Pow failed"
    
    # Test square
    result = gpu.square(a)
    print("square(a):", result.to_list())
    assert result.to_list() == [1.0, 4.0, 9.0, 16.0], "Square failed"
    
    # Test sqrt
    result = gpu.sqrt(a)
    print("sqrt(a):", result.to_list())
    expected = [1.0, math.sqrt(2), math.sqrt(3), 2.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sqrt failed"
    
    # Test exp
    result = gpu.exp(gpu.array([0.0, 1.0, 2.0]))
    print("exp([0, 1, 2]):", result.to_list())
    expected = [1.0, math.e, math.e**2]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Exp failed"
    
    # Test log
    result = gpu.log(gpu.array([1.0, math.e, math.e**2]))
    print("log([1, e, e^2]):", result.to_list())
    expected = [0.0, 1.0, 2.0]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Log failed"
    
    # Test sin
    result = gpu.sin(gpu.array([0.0, math.pi/2, math.pi]))
    print("sin([0, π/2, π]):", result.to_list())
    expected = [0.0, 1.0, 0.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sin failed"
    
    # Test cos
    result = gpu.cos(gpu.array([0.0, math.pi/2, math.pi]))
    print("cos([0, π/2, π]):", result.to_list())
    expected = [1.0, 0.0, -1.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Cos failed"
    
    # Test tan
    result = gpu.tan(gpu.array([0.0, math.pi/4]))
    print("tan([0, π/4]):", result.to_list())
    expected = [0.0, 1.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Tan failed"
    
    # Test asin
    result = gpu.asin(gpu.array([0.0, 0.5, 1.0]))
    print("asin([0, 0.5, 1]):", result.to_list())
    expected = [0.0, math.asin(0.5), math.pi/2]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Asin failed"
    
    # Test atan2
    y = gpu.array([1.0, 1.0, -1.0])
    x = gpu.array([1.0, 0.0, 0.0])
    result = gpu.atan2(y, x)
    print("atan2(y, x):", result.to_list())
    expected = [math.pi/4, math.pi/2, -math.pi/2]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Atan2 failed"
    
    # Test floor
    result = gpu.floor(gpu.array([1.2, 2.7, -1.5]))
    print("floor([1.2, 2.7, -1.5]):", result.to_list())
    assert result.to_list() == [1.0, 2.0, -2.0], "Floor failed"
    
    # Test ceil
    result = gpu.ceil(gpu.array([1.2, 2.7, -1.5]))
    print("ceil([1.2, 2.7, -1.5]):", result.to_list())
    assert result.to_list() == [2.0, 3.0, -1.0], "Ceil failed"
    
    # Test sign
    result = gpu.sign(gpu.array([-5.0, 0.0, 3.0]))
    print("sign([-5, 0, 3]):", result.to_list())
    assert result.to_list() == [-1.0, 0.0, 1.0], "Sign failed"
    
    # Test clip
    result = gpu.clip(gpu.array([1.0, 5.0, 10.0, 15.0]), 3.0, 12.0)
    print("clip([1, 5, 10, 15], 3, 12):", result.to_list())
    assert result.to_list() == [3.0, 5.0, 10.0, 12.0], "Clip failed"
    
    # Test round
    result = gpu.round(gpu.array([1.4, 1.5, 2.5, 3.7]))
    print("round([1.4, 1.5, 2.5, 3.7]):", result.to_list())
    # Note: round uses banker's rounding (round half to even)
    
    # Test broadcast_add
    a_broad = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # length 6
    b_broad = gpu.array([10.0, 20.0])  # length 2
    result = gpu.broadcast_add(a_broad, b_broad)
    print("broadcast_add([1,2,3,4,5,6], [10,20]):", result.to_list())
    assert result.to_list() == [11.0, 22.0, 13.0, 24.0, 15.0, 26.0], "Broadcast add failed"
    
    # Test broadcast_multiply
    result = gpu.broadcast_multiply(a_broad, b_broad)
    print("broadcast_multiply([1,2,3,4,5,6], [10,20]):", result.to_list())
    assert result.to_list() == [10.0, 40.0, 30.0, 80.0, 50.0, 120.0], "Broadcast multiply failed"
    
    print("✅ All elementwise math tests passed!")


def test_activations():
    print("\n=== Testing Category 2: Activations ===")
    gpu = MetalPy()
    
    # Test ReLU
    a = gpu.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = gpu.relu(a)
    print("relu([-2, -1, 0, 1, 2]):", result.to_list())
    assert result.to_list() == [0.0, 0.0, 0.0, 1.0, 2.0], "ReLU failed"
    
    # Test Leaky ReLU - USE TOLERANCE
    result = gpu.leaky_relu(a, alpha=0.1)
    print("leaky_relu([-2, -1, 0, 1, 2], α=0.1):", result.to_list())
    expected = [-0.2, -0.1, 0.0, 1.0, 2.0]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Leaky ReLU failed"
    
    # Test Sigmoid
    result = gpu.sigmoid(gpu.array([0.0, 1.0, -1.0]))
    print("sigmoid([0, 1, -1]):", result.to_list())
    expected = [0.5, 1/(1+math.e**-1), 1/(1+math.e**1)]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Sigmoid failed"
    
    # Test Tanh
    result = gpu.tanh(gpu.array([0.0, 1.0, -1.0]))
    print("tanh([0, 1, -1]):", result.to_list())
    expected = [0.0, math.tanh(1.0), math.tanh(-1.0)]
    assert all(abs(r - e) < 1e-6 for r, e in zip(result.to_list(), expected)), "Tanh failed"
    
    # Test Softplus
    result = gpu.softplus(gpu.array([0.0, 1.0, 2.0]))
    print("softplus([0, 1, 2]):", result.to_list())
    expected = [math.log(2), math.log(1 + math.e), math.log(1 + math.e**2)]
    assert all(abs(r - e) < 1e-5 for r, e in zip(result.to_list(), expected)), "Softplus failed"
    
    # Test Swish
    result = gpu.swish(gpu.array([0.0, 1.0, 2.0]))
    print("swish([0, 1, 2]):", result.to_list())
    # swish(x) = x * sigmoid(x)
    
    # Test GELU
    result = gpu.gelu(gpu.array([0.0, 1.0, -1.0]))
    print("gelu([0, 1, -1]):", result.to_list())
    # GELU is complex, just verify it runs
    
    print("✅ All activation tests passed!")

def test_reductions():
    print("\n=== Testing Category 3: Reductions ===")
    gpu = MetalPy()
    
    # Test sum
    a = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = gpu.sum(a)
    print(f"sum([1,2,3,4,5]): {result}")
    assert abs(result - 15.0) < 1e-5, "Sum failed"
    
    # Test sum on large array
    large = gpu.array([1.0] * 10000)
    result = gpu.sum(large)
    print(f"sum([1.0] * 10000): {result}")
    assert abs(result - 10000.0) < 1e-3, "Large sum failed"
    
    # Test product
    a = gpu.array([2.0, 3.0, 4.0])
    result = gpu.product(a)
    print(f"product([2,3,4]): {result}")
    assert abs(result - 24.0) < 1e-5, "Product failed"
    
    # Test max
    a = gpu.array([3.0, 7.0, 2.0, 9.0, 1.0])
    result = gpu.max(a)
    print(f"max([3,7,2,9,1]): {result}")
    assert abs(result - 9.0) < 1e-5, "Max failed"
    
    # Test min
    result = gpu.min(a)
    print(f"min([3,7,2,9,1]): {result}")
    assert abs(result - 1.0) < 1e-5, "Min failed"
    
    # Test argmax
    result = gpu.argmax(a)
    print(f"argmax([3,7,2,9,1]): {result}")
    assert result == 3, "Argmax failed"
    
    # Test argmin
    result = gpu.argmin(a)
    print(f"argmin([3,7,2,9,1]): {result}")
    assert result == 4, "Argmin failed"
    
    # Test mean
    a = gpu.array([2.0, 4.0, 6.0, 8.0])
    result = gpu.mean(a)
    print(f"mean([2,4,6,8]): {result}")
    assert abs(result - 5.0) < 1e-5, "Mean failed"
    
    print("✅ All reduction tests passed!")


def test_linear_algebra():
    print("\n=== Testing Category 4: Linear Algebra ===")
    gpu = MetalPy()
    
    # Test Hadamard (element-wise matrix multiply)
    A = gpu.array([1.0, 2.0, 3.0, 4.0])  # 2x2 matrix
    B = gpu.array([5.0, 6.0, 7.0, 8.0])  # 2x2 matrix
    result = gpu.hadamard(A, B, rows=2, cols=2)
    print(f"hadamard([[1,2],[3,4]], [[5,6],[7,8]]): {result.to_list()}")
    assert result.to_list() == [5.0, 12.0, 21.0, 32.0], "Hadamard failed"
    
    # Test Matrix multiplication (2x3 @ 3x2 = 2x2)
    # A = [[1, 2, 3],
    #      [4, 5, 6]]  (2x3)
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # B = [[7,  8],
    #      [9, 10],
    #      [11, 12]]  (3x2)
    B = gpu.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    
    result = gpu.matmul(A, B, M=2, N=2, K=3)
    print(f"matmul(2x3, 3x2): {result.to_list()}")
    # Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12],
    #            [4*7+5*9+6*11, 4*8+5*10+6*12]]
    #         = [[58, 64], [139, 154]]
    expected = [58.0, 64.0, 139.0, 154.0]
    assert all(abs(r - e) < 1e-4 for r, e in zip(result.to_list(), expected)), "Matmul failed"
    
    # Test Transpose
    # A = [[1, 2, 3],
    #      [4, 5, 6]]  (2x3)
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = gpu.transpose(A, rows=2, cols=3)
    print(f"transpose(2x3): {result.to_list()}")
    # Expected: [[1, 4],
    #            [2, 5],
    #            [3, 6]]  (3x2, row-major)
    expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    assert result.to_list() == expected, "Transpose failed"
    
    # Test Row sum
    # A = [[1, 2, 3],
    #      [4, 5, 6]]
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = gpu.row_sum(A, rows=2, cols=3)
    print(f"row_sum([[1,2,3],[4,5,6]]): {result.to_list()}")
    assert result.to_list() == [6.0, 15.0], "Row sum failed"
    
    # Test Column sum
    result = gpu.col_sum(A, rows=2, cols=3)
    print(f"col_sum([[1,2,3],[4,5,6]]): {result.to_list()}")
    assert result.to_list() == [5.0, 7.0, 9.0], "Col sum failed"
    
    # Test Row scale
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3
    s = gpu.array([2.0, 3.0])  # scale factors for 2 rows
    result = gpu.row_scale(A, s, rows=2, cols=3)
    print(f"row_scale([[1,2,3],[4,5,6]], [2,3]): {result.to_list()}")
    expected = [2.0, 4.0, 6.0, 12.0, 15.0, 18.0]
    assert result.to_list() == expected, "Row scale failed"
    
    # Test Column scale
    A = gpu.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 2x3
    s = gpu.array([10.0, 100.0, 1000.0])  # scale factors for 3 cols
    result = gpu.col_scale(A, s, rows=2, cols=3)
    print(f"col_scale([[1,2,3],[4,5,6]], [10,100,1000]): {result.to_list()}")
    expected = [10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]
    assert result.to_list() == expected, "Col scale failed"
    
    print("✅ All linear algebra tests passed!")


def test_miscellaneous():
    print("\n=== Testing Category 5: Miscellaneous ===")
    gpu = MetalPy()
    
    # Test slice
    a = gpu.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    result = gpu.slice(a, start=1, end=4)
    print(f"slice([10,20,30,40,50,60], 1:4): {result.to_list()}")
    assert result.to_list() == [20.0, 30.0, 40.0], "Slice failed"
    
    result = gpu.slice(a, start=0, end=2)
    print(f"slice([10,20,30,40,50,60], 0:2): {result.to_list()}")
    assert result.to_list() == [10.0, 20.0], "Slice failed"
    
    # Test bitonic sort
    # Must be power of 2
    a = gpu.array([64.0, 2.0, 25.0, 12.0, 22.0, 11.0, 90.0, 8.0])
    print(f"Before sort: {a.to_list()}")
    result = gpu.bitonic_sort(a)
    print(f"After bitonic_sort: {result.to_list()}")
    sorted_vals = result.to_list()
    assert sorted_vals == sorted(sorted_vals), "Bitonic sort failed"
    
    # Test with powers of 2 array
    a = gpu.array([5.0, 1.0, 9.0, 3.0])
    result = gpu.bitonic_sort(a)
    print(f"bitonic_sort([5,1,9,3]): {result.to_list()}")
    assert result.to_list() == [1.0, 3.0, 5.0, 9.0], "Bitonic sort failed"
    
    print("✅ All miscellaneous tests passed!")


def test_edge_cases():
    print("\n=== Testing Edge Cases ===")
    gpu = MetalPy()
    
    # Single element
    a = gpu.array([42.0])
    result = gpu.sum(a)
    print(f"sum([42]): {result}")
    assert abs(result - 42.0) < 1e-5, "Single element sum failed"
    
    # Large array reduction
    n = 1000000
    large = gpu.array([1.0] * n)
    result = gpu.sum(large)
    print(f"sum([1.0] * {n}): {result}")
    assert abs(result - float(n)) < 100, "Large array sum failed"
    
    # Negative numbers
    a = gpu.array([-5.0, -10.0, -15.0])
    result = gpu.sum(a)
    print(f"sum([-5,-10,-15]): {result}")
    assert abs(result - (-30.0)) < 1e-5, "Negative sum failed"
    
    print("✅ All edge case tests passed!")


def main():
    print("🚀 Starting MetalPy Test Suite\n")
    
    try:
        test_elementwise_math()
        test_activations()
        test_reductions()
        test_linear_algebra()
        test_miscellaneous()
        test_edge_cases()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*50)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()