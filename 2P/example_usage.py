"""
Example usage of the MatrixSolver class
"""

import numpy as np
from matrix import MatrixSolver

# Set NumPy print options for better formatting
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

def example_1():
    """Example 1: Simple 2x2 system"""
    print("=== Example 1: Simple 2x2 System ===")
    
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 6])
    
    solver = MatrixSolver(A, b)
    
    # LU Factorization
    L, U = solver.lu_factorization()
    x_lu = solver.solve_lu(L, U)
    print(f"LU Solution: {x_lu}")
    
    # Jacobi Method
    x_jacobi, iters, residuals = solver.jacobi_method(tol=1e-6)
    print(f"Jacobi Solution: {x_jacobi} (iterations: {iters})")
    
    # Gauss-Seidel Method
    x_gs, iters, residuals = solver.gauss_seidel_method(tol=1e-6)
    print(f"Gauss-Seidel Solution: {x_gs} (iterations: {iters})")
    
    # Verify with NumPy
    x_numpy = np.linalg.solve(A, b)
    print(f"NumPy Solution: {x_numpy}")
    print()

def example_2():
    """Example 2: Larger system with different convergence parameters"""
    print("=== Example 2: 4x4 System with Custom Parameters ===")
    
    # Create a diagonally dominant matrix
    A = np.array([
        [10, -1, 2, 0],
        [-1, 11, -1, 3],
        [2, -1, 10, -1],
        [0, 3, -1, 8]
    ])
    b = np.array([6, 25, -11, 15])
    
    solver = MatrixSolver(A, b)
    
    # Test with different initial guesses
    x0_1 = np.zeros(4)
    x0_2 = np.ones(4)
    x0_3 = np.array([1, -1, 1, -1])
    
    print("Testing Jacobi method with different initial guesses:")
    for i, x0 in enumerate([x0_1, x0_2, x0_3], 1):
        x, iters, residuals = solver.jacobi_method(x0=x0, tol=1e-8, max_iter=500)
        print(f"  Initial guess {i}: {x0} -> Solution: {x} (iterations: {iters})")
    
    print("\nTesting Gauss-Seidel with different tolerances:")
    for tol in [1e-4, 1e-6, 1e-8]:
        x, iters, residuals = solver.gauss_seidel_method(tol=tol)
        print(f"  Tolerance {tol}: {iters} iterations, residual: {residuals[-1]:.2e}")
    print()

def example_3():
    """Example 3: Performance comparison"""
    print("=== Example 3: Performance Comparison ===")
    
    # Create a larger system
    n = 10
    A = np.random.rand(n, n)
    # Make it diagonally dominant to ensure convergence
    A = A + n * np.eye(n)
    b = np.random.rand(n)
    
    solver = MatrixSolver(A, b)
    
    print(f"Solving {n}x{n} system...")
    
    # LU Factorization
    import time
    start = time.time()
    L, U = solver.lu_factorization()
    x_lu = solver.solve_lu(L, U)
    lu_time = time.time() - start
    
    # Jacobi Method
    start = time.time()
    x_jacobi, iters_j, residuals_j = solver.jacobi_method(tol=1e-6)
    jacobi_time = time.time() - start
    
    # Gauss-Seidel Method
    start = time.time()
    x_gs, iters_gs, residuals_gs = solver.gauss_seidel_method(tol=1e-6)
    gs_time = time.time() - start
    
    print(f"LU Factorization: {lu_time:.4f}s, residual: {solver.check_solution(x_lu):.2e}")
    print(f"Jacobi Method: {jacobi_time:.4f}s, {iters_j} iterations, residual: {residuals_j[-1]:.2e}")
    print(f"Gauss-Seidel: {gs_time:.4f}s, {iters_gs} iterations, residual: {residuals_gs[-1]:.2e}")
    print()

def example_4():
    """Example 4: Error handling"""
    print("=== Example 4: Error Handling ===")
    
    # Matrix with zero diagonal (will cause issues with iterative methods)
    A = np.array([[0, 1], [1, 0]])
    b = np.array([1, 2])
    
    try:
        solver = MatrixSolver(A, b)
        x_jacobi, iters, residuals = solver.jacobi_method()
        print(f"Jacobi solution: {x_jacobi}")
    except ValueError as e:
        print(f"Jacobi method error: {e}")
    
    try:
        solver = MatrixSolver(A, b)
        x_gs, iters, residuals = solver.gauss_seidel_method()
        print(f"Gauss-Seidel solution: {x_gs}")
    except ValueError as e:
        print(f"Gauss-Seidel method error: {e}")
    
    # LU factorization should work for this matrix
    try:
        solver = MatrixSolver(A, b)
        L, U = solver.lu_factorization()
        x_lu = solver.solve_lu(L, U)
        print(f"LU solution: {x_lu}")
    except Exception as e:
        print(f"LU factorization error: {e}")
    print()

if __name__ == "__main__":
    example_1()
    example_2()
    example_3()
    example_4() 