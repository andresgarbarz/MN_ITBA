# Matrix Solver - Numerical Methods Implementation

This script implements three fundamental numerical methods for solving linear systems of equations:

1. **LU Factorization (Decomposition)**
2. **Jacobi Method**
3. **Gauss-Seidel Method**

## Features

- **LU Factorization**: Direct method that decomposes matrix A into L (lower triangular) and U (upper triangular) matrices
- **Jacobi Method**: Iterative method that uses diagonal dominance for convergence
- **Gauss-Seidel Method**: Iterative method that typically converges faster than Jacobi
- **Error Handling**: Robust error checking for edge cases
- **Performance Comparison**: Built-in timing and convergence analysis
- **Flexible Parameters**: Customizable tolerance, maximum iterations, and initial guesses

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:

- numpy >= 1.21.0
- matplotlib >= 3.5.0 (for plotting residuals if needed)
- sympy >= 1.9.0

## Usage

### Basic Usage

```python
import numpy as np
from matrix import MatrixSolver

# Define your system Ax = b
A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
b = np.array([1, 5, 0])

# Create solver instance
solver = MatrixSolver(A, b)

# Solve using LU factorization
L, U = solver.lu_factorization()
x_lu = solver.solve_lu(L, U)

# Solve using Jacobi method
x_jacobi, iterations, residuals = solver.jacobi_method(tol=1e-6)

# Solve using Gauss-Seidel method
x_gs, iterations, residuals = solver.gauss_seidel_method(tol=1e-6)
```

### Running the Demo

```bash
python matrix.py
```

This will run a comprehensive demo showing all three methods on a sample 3x3 system.

## Method Details

### 1. LU Factorization

**Algorithm**: Decomposes A = L × U where:

- L is lower triangular with 1's on diagonal
- U is upper triangular

**Advantages**:

- Direct method (no iterations needed)
- Can solve multiple right-hand sides efficiently
- Numerically stable for most matrices

**Limitations**:

- Requires matrix to be non-singular
- May fail with zero pivots

### 2. Jacobi Method

**Algorithm**: Iterative method using the formula:

```
x^(k+1) = D^(-1) × (b - R × x^(k))
```

where D is the diagonal of A and R = A - D.

**Advantages**:

- Simple to implement
- Parallelizable
- Works well for diagonally dominant matrices

**Limitations**:

- Requires diagonal elements to be non-zero
- May converge slowly for some matrices
- Requires diagonal dominance for guaranteed convergence

### 3. Gauss-Seidel Method

**Algorithm**: Iterative method that uses updated values immediately:

```
x_i^(k+1) = (b_i - Σ_{j≠i} a_{ij} × x_j^(k+1 or k)) / a_{ii}
```

**Advantages**:

- Typically converges faster than Jacobi
- Uses updated values as soon as they're available
- Good for diagonally dominant matrices

**Limitations**:

- Not easily parallelizable
- Requires diagonal elements to be non-zero
- Requires diagonal dominance for guaranteed convergence

## Parameters

### Common Parameters

- `x0`: Initial guess vector (default: zero vector)
- `max_iter`: Maximum number of iterations (default: 1000)
- `tol`: Convergence tolerance (default: 1e-6)

### Example with Custom Parameters

```python
# Custom initial guess and parameters
x0 = np.array([1, 1, 1])
x, iterations, residuals = solver.jacobi_method(
    x0=x0,
    max_iter=500,
    tol=1e-8
)
```

## Error Handling

The script includes comprehensive error handling for:

- **Non-square matrices**: Raises ValueError
- **Dimension mismatches**: Raises ValueError
- **Zero diagonal elements**: Raises ValueError for iterative methods
- **Zero pivots**: Raises ValueError for LU factorization
- **Non-convergence**: Prints warning and returns best approximation

## Performance Notes

- **LU Factorization**: O(n³) complexity, best for small to medium matrices
- **Jacobi Method**: O(n²) per iteration, good for sparse matrices
- **Gauss-Seidel**: O(n²) per iteration, typically fewer iterations than Jacobi

## Convergence Criteria

Both iterative methods use the residual norm as convergence criterion:

```
||Ax - b|| < tolerance
```

## Verification

The script includes verification against NumPy's `np.linalg.solve()` to ensure accuracy. All methods should produce solutions with residuals on the order of machine precision.

## Example Output

```
=== Matrix Solver Demo ===

Matrix A:
[[ 4 -1  0]
 [-1  4 -1]
 [ 0 -1  4]]

Vector b:
[1 5 0]

1. LU FACTORIZATION
--------------------
L matrix:
[[ 1.          0.          0.        ]
 [-0.25        1.          0.        ]
 [ 0.         -0.26666667  1.        ]]

U matrix:
[[ 4.         -1.          0.        ]
 [ 0.          3.75       -1.        ]
 [ 0.          0.          3.73333333]]

Solution (LU): [0.625 1.5   0.375]
Residual: 2.22e-16

2. JACOBI METHOD
--------------------
Solution (Jacobi): [0.625 1.5   0.375]
Iterations: 20
Final residual: 4.70e-09
```

## Contributing

Feel free to extend the script with additional numerical methods or improvements!
