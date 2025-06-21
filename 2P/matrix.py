import numpy as np
from typing import Tuple, List, Optional
import copy

class MatrixSolver:
    """
    A class that implements various numerical methods for solving linear systems:
    1. LU Factorization (Decomposition)
    2. Jacobi Method
    3. Gauss-Seidel Method
    """

    # Set NumPy print options for better formatting
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
    
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Initialize the solver with coefficient matrix A and constant vector b.
        
        Args:
            A: Coefficient matrix (n x n)
            b: Constant vector (n x 1)
        """
        self.A = A.astype(float)
        self.b = b.astype(float).flatten()
        self.n = A.shape[0]
        
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensions of A and b must match")
    
    def lu_factorization(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform LU factorization of matrix A.
        
        Returns:
            Tuple of (L, U) matrices where A = L * U
        """
        A = self.A.copy()
        n = self.n
        
        # Initialize L as identity matrix and U as copy of A
        L = np.eye(n)
        U = A.copy()
        
        for k in range(n-1):
            # Check if pivot is zero
            if abs(U[k, k]) < 1e-10:
                raise ValueError(f"Zero pivot encountered at position ({k}, {k})")
            
            for i in range(k+1, n):
                # Compute multiplier
                multiplier = U[i, k] / U[k, k]
                L[i, k] = multiplier
                
                # Update U matrix
                for j in range(k, n):
                    U[i, j] -= multiplier * U[k, j]
        
        return L, U
    
    def solve_lu(self, L: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Solve the system using LU factorization.
        
        Args:
            L: Lower triangular matrix
            U: Upper triangular matrix
            
        Returns:
            Solution vector x
        """
        # Solve Ly = b (forward substitution)
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = self.b[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
            y[i] /= L[i, i]
        
        # Solve Ux = y (backward substitution)
        x = np.zeros(self.n)
        for i in range(self.n-1, -1, -1):
            x[i] = y[i]
            for j in range(i+1, self.n):
                x[i] -= U[i, j] * x[j]
            x[i] /= U[i, i]
        
        return x
    
    def jacobi_method(self, x0: Optional[np.ndarray] = None,
                        max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, int, List[float]]:
        """
        Solve the system using Jacobi iterative method.
        
        Args:
            x0: Initial guess (default: zero vector)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Tuple of (solution, iterations, residuals)
        """
        if x0 is None:
            x0 = np.zeros(self.n)
        
        x = x0.copy()
        residuals = []
        
        # Extract diagonal and off-diagonal parts
        D = np.diag(np.diag(self.A))
        R = self.A - D
        
        # Check if diagonal elements are non-zero
        if np.any(np.diag(D) == 0):
            raise ValueError("Matrix A has zero diagonal elements")
        
        D_inv = np.diag(1.0 / np.diag(D))
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # Jacobi iteration: x^(k+1) = D^(-1) * (b - R * x^(k))
            x = D_inv @ (self.b - R @ x_old)
            
            # Calculate residual
            residual = np.linalg.norm(self.A @ x - self.b)
            residuals.append(residual)
            
            # Check convergence
            if residual < tol:
                return x, iteration + 1, residuals
        
        print(f"Warning: Jacobi method did not converge after {max_iter} iterations")
        return x, max_iter, residuals
    
    def gauss_seidel_method(self, x0: Optional[np.ndarray] = None,
                        max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, int, List[float]]:
        """
        Solve the system using Gauss-Seidel iterative method.
        
        Args:
            x0: Initial guess (default: zero vector)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Tuple of (solution, iterations, residuals)
        """
        if x0 is None:
            x0 = np.zeros(self.n)
        
        x = x0.copy()
        residuals = []
        
        # Extract diagonal, lower triangular, and upper triangular parts
        D = np.diag(np.diag(self.A))
        L = np.tril(self.A, k=-1)  # Lower triangular (excluding diagonal)
        U = np.triu(self.A, k=1)   # Upper triangular (excluding diagonal)
        
        # Check if diagonal elements are non-zero
        if np.any(np.diag(D) == 0):
            raise ValueError("Matrix A has zero diagonal elements")
        
        # Pre-compute (D + L)^(-1) for efficiency
        D_plus_L = D + L
        D_plus_L_inv = np.linalg.inv(D_plus_L)
        
        for iteration in range(max_iter):
            x_old = x.copy()
            
            # Gauss-Seidel iteration: x^(k+1) = (D + L)^(-1) * (b - U * x^(k))
            x = D_plus_L_inv @ (self.b - U @ x_old)
            
            # Calculate residual
            residual = np.linalg.norm(self.A @ x - self.b)
            residuals.append(residual)
            
            # Check convergence
            if residual < tol:
                return x, iteration + 1, residuals
        
        print(f"Warning: Gauss-Seidel method did not converge after {max_iter} iterations")
        return x, max_iter, residuals
    
    def check_solution(self, x: np.ndarray) -> float:
        """
        Check the accuracy of a solution by calculating the residual.
        
        Args:
            x: Solution vector
            
        Returns:
            Residual norm ||Ax - b||
        """
        return np.linalg.norm(self.A @ x - self.b)


def demo():
    """
    Demonstration of all three methods with a simple example.
    """
    # Set NumPy print options for better formatting
    np.set_printoptions(precision=4, suppress=True, floatmode='fixed')
    
    print("=== Matrix Solver Demo ===\n")
    
    # Example matrix and vector
    A = np.array([
        [4, 3, -1],
        [-2, -4, 5],
        [1, 2, 6]
    ])
    b = np.array([1, 5, 0])
    
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
    print("\n" + "="*50)
    
    # Create solver instance
    solver = MatrixSolver(A, b)
    
    # 1. LU Factorization
    print("\n1. LU FACTORIZATION")
    print("-" * 20)
    try:
        L, U = solver.lu_factorization()
        print("L matrix:")
        print(L)
        print("\nU matrix:")
        print(U)
        print("\nVerification: L * U = A")
        print(L @ U)
        
        x_lu = solver.solve_lu(L, U)
        print(f"\nSolution (LU): {x_lu}")
        print(f"Residual: {solver.check_solution(x_lu):.2e}")
        
    except Exception as e:
        print(f"LU factorization failed: {e}")
    
    # 2. Jacobi Method
    print("\n\n2. JACOBI METHOD")
    print("-" * 20)
    try:
        x_jacobi, iterations, residuals = solver.jacobi_method(tol=1e-8)
        print(f"Solution (Jacobi): {x_jacobi}")
        print(f"Iterations: {iterations}")
        print(f"Final residual: {residuals[-1]:.2e}")
        print(f"Residual: {solver.check_solution(x_jacobi):.2e}")
        
    except Exception as e:
        print(f"Jacobi method failed: {e}")
    
    # 3. Gauss-Seidel Method
    print("\n\n3. GAUSS-SEIDEL METHOD")
    print("-" * 20)
    try:
        x_gs, iterations, residuals = solver.gauss_seidel_method(tol=1e-8)
        print(f"Solution (Gauss-Seidel): {x_gs}")
        print(f"Iterations: {iterations}")
        print(f"Final residual: {residuals[-1]:.2e}")
        print(f"Residual: {solver.check_solution(x_gs):.2e}")
        
    except Exception as e:
        print(f"Gauss-Seidel method failed: {e}")
    
    # Compare solutions
    print("\n\n4. COMPARISON")
    print("-" * 20)
    try:
        x_numpy = np.linalg.solve(A, b)
        print(f"NumPy solution: {x_numpy}")
        print(f"NumPy residual: {solver.check_solution(x_numpy):.2e}")
        
        if 'x_lu' in locals():
            print(f"LU vs NumPy difference: {np.linalg.norm(x_lu - x_numpy):.2e}")
        if 'x_jacobi' in locals():
            print(f"Jacobi vs NumPy difference: {np.linalg.norm(x_jacobi - x_numpy):.2e}")
        if 'x_gs' in locals():
            print(f"Gauss-Seidel vs NumPy difference: {np.linalg.norm(x_gs - x_numpy):.2e}")
            
    except Exception as e:
        print(f"Comparison failed: {e}")

def ejercicio_teoria():
    A = np.array([
        [4, 3, -1],
        [-2, -4, 5],
        [1, 2, 6]
    ])
    b = np.array([1, 5, 0])
    solver = MatrixSolver(A, b)
    L, U = solver.lu_factorization()
    print("L matrix:")
    print(L)
    print("\nU matrix:")
    print(U)
    # x = solver.solve_lu(L, U)
    # print(x)

def ejercicio_1():
    A = np.array([
        [4, 2, 1],
        [1, 3, 1],
        [3, 2, 6]
    ])
    b = np.array([4, 4, 7])
    solver = MatrixSolver(A, b)
    x, iterations, residuals = solver.jacobi_method(x0=np.array([0.1, 0.8, 0.5]), max_iter=3)
    print("Jacobi solution:")
    print(x)

def ejercicio_2():    
    A = np.array([
        [5**2, 5, 1],
        [8**2, 8, 1],
        [12**2, 12, 1]
    ])
    b = np.array([106.8, 177.2, 279.2])
    solver = MatrixSolver(A, b)
    x, iterations, residuals = solver.gauss_seidel_method(x0=np.array([1, 2, 3]), max_iter=2)
    print("Gauss-Seidel solution:")
    print(x)


if __name__ == "__main__":
    ejercicio_teoria()
    ejercicio_1()
    ejercicio_2()